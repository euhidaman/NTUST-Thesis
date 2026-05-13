"""
REST API Server for EmberVLM

FastAPI-based server for inference and robot selection.
"""

import os
import io
import base64
from typing import Optional, List
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import torch
from PIL import Image


# Request/Response models
if FASTAPI_AVAILABLE:
    class AnalyzeRequest(BaseModel):
        """Request for incident analysis."""
        image_base64: Optional[str] = None
        instruction: Optional[str] = None

    class RobotSelection(BaseModel):
        """Robot selection result."""
        selected_robot: str
        robot_index: int
        confidence: float
        robot_probabilities: dict
        action_plan: str
        latency_ms: float

    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        model_loaded: bool
        device: str


class EmberVLMServer:
    """
    EmberVLM API Server.

    Provides REST endpoints for:
    - Incident analysis
    - Robot selection
    - Model info
    """

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")

        self.model_path = model_path
        self.host = host
        self.port = port

        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create FastAPI app
        self.app = FastAPI(
            title="EmberVLM API",
            description="Tiny Multimodal VLM for Robot Fleet Selection",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""

        @self.app.on_event("startup")
        async def startup():
            """Load model on startup."""
            self._load_model()

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                device=self.device,
            )

        @self.app.post("/analyze", response_model=RobotSelection)
        async def analyze(
            image: Optional[UploadFile] = File(None),
            instruction: Optional[str] = None,
        ):
            """
            Analyze incident and select robot.

            Upload an image and optionally provide an instruction.
            Returns robot selection with confidence and action plan.
            """
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            if image is None:
                raise HTTPException(status_code=400, detail="Image required")

            # Read and process image
            try:
                image_bytes = await image.read()
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

            # Preprocess
            pixel_values = self._preprocess_image(pil_image)

            # Run inference
            import time
            start = time.perf_counter()

            result = self.model.analyze_incident(
                pixel_values=pixel_values,
                instruction=instruction or "Analyze this incident and select the best robot.",
                tokenizer=self.tokenizer,
            )

            latency = (time.perf_counter() - start) * 1000

            return RobotSelection(
                selected_robot=result['selected_robot'],
                robot_index=result['robot_index'],
                confidence=result['confidence'],
                robot_probabilities=result['robot_probabilities'],
                action_plan=result.get('action_plan', ''),
                latency_ms=latency,
            )

        @self.app.post("/analyze_base64", response_model=RobotSelection)
        async def analyze_base64(request: AnalyzeRequest):
            """
            Analyze incident from base64-encoded image.
            """
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            if not request.image_base64:
                raise HTTPException(status_code=400, detail="image_base64 required")

            # Decode image
            try:
                image_bytes = base64.b64decode(request.image_base64)
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

            # Preprocess
            pixel_values = self._preprocess_image(pil_image)

            # Run inference
            import time
            start = time.perf_counter()

            result = self.model.analyze_incident(
                pixel_values=pixel_values,
                instruction=request.instruction or "Analyze this incident.",
                tokenizer=self.tokenizer,
            )

            latency = (time.perf_counter() - start) * 1000

            return RobotSelection(
                selected_robot=result['selected_robot'],
                robot_index=result['robot_index'],
                confidence=result['confidence'],
                robot_probabilities=result['robot_probabilities'],
                action_plan=result.get('action_plan', ''),
                latency_ms=latency,
            )

        @self.app.get("/robots")
        async def get_robots():
            """Get available robot fleet."""
            return {
                "robots": [
                    {
                        "name": "Drone",
                        "capabilities": ["aerial_survey", "reconnaissance"],
                        "terrain": ["air"],
                    },
                    {
                        "name": "Humanoid",
                        "capabilities": ["manipulation", "tool_use"],
                        "terrain": ["indoor", "outdoor_flat"],
                    },
                    {
                        "name": "Wheeled",
                        "capabilities": ["heavy_transport", "long_range"],
                        "terrain": ["road", "indoor"],
                    },
                    {
                        "name": "Legged",
                        "capabilities": ["rough_terrain", "climbing"],
                        "terrain": ["rocky", "stairs", "slopes"],
                    },
                    {
                        "name": "Underwater",
                        "capabilities": ["aquatic_ops", "diving"],
                        "terrain": ["water"],
                    },
                ]
            }

        @self.app.get("/model_info")
        async def model_info():
            """Get model information."""
            if self.model is None:
                return {"status": "not_loaded"}

            params = self.model.count_parameters()

            return {
                "status": "loaded",
                "model_name": "EmberVLM",
                "total_parameters": params['total'],
                "trainable_parameters": params['trainable'],
                "device": self.device,
            }

    def _load_model(self):
        """Load model and tokenizer."""
        from embervlm.models import EmberVLM
        from transformers import AutoTokenizer

        print(f"Loading model from {self.model_path}")

        self.model = EmberVLM.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        tokenizer_path = Path(self.model_path) / 'tokenizer'
        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.device}")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        pixel_values = transform(image).unsqueeze(0)
        return pixel_values.to(self.device)

    def run(self):
        """Run the server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)


def create_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> EmberVLMServer:
    """
    Create EmberVLM API server.

    Args:
        model_path: Path to model checkpoint
        host: Server host
        port: Server port

    Returns:
        EmberVLMServer instance
    """
    return EmberVLMServer(model_path, host, port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EmberVLM API Server")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port')

    args = parser.parse_args()

    server = create_server(args.model_path, args.host, args.port)
    server.run()


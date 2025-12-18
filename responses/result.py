from pydantic import BaseModel, Field
from typing import Dict, Optional

class Result(BaseModel):
    id: str
    message: str = Field(default='Request accepted')
    status: str = Field(default='pending')
    comfyui_job_id: Optional[str] = Field(default=None, description="ComfyUI prompt_id for cancel support")
    comfyui_response: Dict = Field(default={})
    output: list = Field(default=[])
    timings: Dict = Field(default={})


from pydantic import BaseModel, Field
from typing import Any
from tools.base import Tool, ToolResult, ToolInvocation, ToolKind
import asyncio

class STTStreamParams(BaseModel):
    audio: Any = Field(..., description="Audio chunk as numpy array")
    rate: int = Field(..., description="Sample rate (e.g., 16000)")

class SpeechToTextStreamTool(Tool):

    name = "speech_to_text_stream"
    description = "Convert live audio chunks into text using STT"
    kind = ToolKind.READ

    @property
    def schema(self):
        return STTStreamParams
    
    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        try:
            errors = self.validate_params(invocation.params)
            if errors:
                return ToolResult.error_result(", ".join(errors))

            params = STTStreamParams(**invocation.params)

            engine = self.config.stt_engine

            processed = engine.processor.process(params.audio, params.rate)
            audio_bytes = engine.processor.to_bytes(processed)

           
            text = await asyncio.to_thread(
                engine.provider.transcribe,
                audio_bytes
            )

            return ToolResult.success_result(
                output=text,
                metadata={"text": text}
            )

        except Exception as e:
            return ToolResult.error_result(
                error=f"[STT TOOL ERROR] {str(e)}"
            )

       
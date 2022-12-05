from pathlib import Path
import ffmpeg
from tap import Tap
from stable_whisper import load_model


class Args(Tap):
    audio: str  # Your audio file
    output: str = "segments"  # Where the output segments will be written to
    model: str = "medium.en"  # The Whisper model


if __name__ == "__main__":
    args = Args().parse_args()

    model = load_model(args.model)
    segments = model.transcribe(
        args.audio,
        top_focus=True,
        pbar=True,
        language="en",
        suppress_silence=False,
    )["segments"]

    segment_dir = Path(args.output)
    segment_dir.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        (
            ffmpeg.input(args.audio, ss=segment["start"])
            .output(
                f"{segment_dir / ('%03d' % segment['id'])}.wav",
                t=segment["end"] - segment["start"],
            )
            .run(quiet=True, overwrite_output=True)
        )

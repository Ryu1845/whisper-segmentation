import argparse
import json
import sys
from pathlib import Path

import ffmpeg
import pywhisper as whisper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a")
    parser.add_argument("-d")
    args = parser.parse_args()
    if args.a is None:
        print("Please provide audio (with -a)")
        sys.exit(1)

    model = whisper.load_model("medium.en")
    result = model.transcribe(args.a)
    segment_dir = Path(args.d or "segments")
    segment_dir.mkdir(parents=True, exist_ok=True)
    for segment in result["segments"]:
        (
            ffmpeg.input(args.a, ss=segment["start"])
            .output(
                f"{segment_dir / ('%03d' % segment['id'])}.wav",
                t=segment["end"] - segment["start"],
            )
            .run()
        )

    print(json.dumps(result["segments"], indent=2))

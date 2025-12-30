
import os
import sys
import logging
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from modules.pipeline import VideoTranslationPipeline, PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def translate_video():
    """Translate video using pipeline"""
    print("=" * 60)
    print("CUSTOM VIDEO TRANSLATION")
    print("=" * 60)

    # Input video path from request
    input_video_path = r"e:\Apps\octavia-assignment\input\AI Engineering Bootcamp _ Master Generative AI with LunarTech _ Lets Build The Future Together.mp4"
    
    if not os.path.exists(input_video_path):
        print(f"❌ Input video not found: {input_video_path}")
        return

    print(f"✅ Input video found: {input_video_path}")

    # Initialize pipeline
    print("\n🚀 Initializing Video Translation Pipeline...")
    config = PipelineConfig(
        chunk_size=30,
        use_gpu=False,
        temp_dir="/tmp/octavia_custom_translation",
        output_dir="backend/outputs"
    )

    pipeline = VideoTranslationPipeline(config)

    target_language = "ru"
    print(f"🎯 Target Language: {target_language}")

    try:
        # Process the video
        result = pipeline.process_video_fast(input_video_path, target_language)

        if result.get("success"):
            print("\n✅ SUCCESS: Video translation completed!")
            print(f"📁 Output: {result.get('output_video', 'N/A')}")
        else:
            print("\n❌ FAILED: Video translation failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    translate_video()

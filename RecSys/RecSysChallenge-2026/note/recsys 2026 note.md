## command
git clone https://github.com/proudSweet/ai-coding-joy.git

cd ./ai-coding-joy/RecSys/RecSysChallenge-2026/music-crs-baselines
pip install -e .
pip install flash-attn --no-build-isolation

python run_inference_devset.py --tid qwen_bm25_devset --batch_size 16
python run_inference_blindset.py --tid qwen_bm25_blindset_A --batch_size 16
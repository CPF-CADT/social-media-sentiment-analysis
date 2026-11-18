# Sentiment pipeline infrastructure

This Terraform module provisions an Ubuntu EC2 host with persistent EBS storage
for the YouTube collection and Gemini labeling workflow.

The instance setup installs:

- `google-api-python-client` for `collector.py`
- `python-dotenv` for `.env` loading
- `google-genai` for the Gemini labeling script
- Python virtual-environment and Chromium support for related collectors

After deployment, copy the project to `var.app_root`, then create the runtime
configuration as the `ubuntu` user:

```bash
cd /opt/social-media-sentiment-analysis
cp .env.example .env
chmod 600 .env
```

Set at least:

```dotenv
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Run the collection stage from the project root:

```bash
source .venv/bin/activate
python collector.py
```

Run the Gemini labeling stage with the project’s Gemini labeler after it is
deployed. Both stages use the persistent `data/` mount, so a restarted or
replaced instance does not lose collected CSV data.

The current Terraform state is local and ignored. Review `terraform plan`
before applying because changing the resource names from the former LegalRAG
configuration can create replacement resources.

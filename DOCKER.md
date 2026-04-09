# Docker + GHCR

This repo now includes:

- `Dockerfile` for the Streamlit app
- `requirements.docker.txt` for the container build
- `.github/workflows/publish-ghcr.yml` for manual publishing to GitHub Container Registry

## Important

The Docker image contains the application code and Python dependencies.
It does **not** include your dataset, trained models, or generated results.
Those should be mounted into the container at runtime.

## Publish from GitHub

This workflow runs **only when you click it manually**:

1. Push this branch to GitHub.
2. Open `Actions` in your GitHub repo.
3. Open `Publish Docker Image`.
4. Click `Run workflow`.
5. Leave `platforms` as `linux/arm64` for your Oracle Ubuntu AArch64 VM.
6. Pick a tag such as `latest` or `oracle-arm64`.

The image will be published to:

`ghcr.io/<owner>/<repo>:<tag>`

## Run on Oracle Ubuntu AArch64

Install Docker on the VM, then:

```bash
docker login ghcr.io
docker pull ghcr.io/<owner>/<repo>:latest
mkdir -p ~/botnet/data ~/botnet/models ~/botnet/results
docker run -d \
  --name botnet-streamlit \
  -p 8501:8501 \
  -v ~/botnet/data:/app/data \
  -v ~/botnet/models:/app/models \
  -v ~/botnet/results:/app/results \
  ghcr.io/<owner>/<repo>:latest
```

Then place your dataset in:

`~/botnet/data/UNSW_2018_IoT_Botnet_Full5pc_4.csv`

## Notes

- If you want the predefined dataset option to work on the VM, that CSV must exist in the mounted `data/` directory.
- If you use the upload option in Streamlit, the uploaded CSV will also land in `/app/data`.
- Because this project uses TensorFlow, `linux/arm64` builds can take a while. That is normal.

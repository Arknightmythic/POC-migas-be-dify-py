# Dokuprime API

API to interact with dokuprime database

## Installation

Create a virtual environment

```bash
python -m venv .venv
```

activate .venv (Windows)

```bash
.venv/Scripts/activate
```

activate .venv (Ubuntu)

```bash
Source .venv/bin/activate
```

then, install the requirements

```bash
pip install -r requirements.txt
```

Run the app by using:

```bash
uvicorn main:app --host 0.0.0.0 --port 9797
```

Or using docker:

```bash
docker compose up -d
```

Note: don't forget to change `localhost` to `host.docker.internal` for local development!
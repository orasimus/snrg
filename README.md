# Testing deployment

```
python3 -m venv .venv
.venv/scripts/activate.bat # Windows
source venv/bin/activate # UNIX
pip install -r requirements.txt
python deploy.py
```

After this, you should have the web server running. You can test it with curl for example like this: `curl -F 'file=@./5.png' http://localhost:8000`

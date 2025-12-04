import os
from dotenv import load_dotenv
from face_check import app

load_dotenv()

# Export app for Vercel - this is required!
# Vercel will call this app object directly
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

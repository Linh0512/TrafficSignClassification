from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return {"message": "Hello from Flask on Vercel!", "status": "working"}

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run() 
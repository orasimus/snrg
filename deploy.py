from werkzeug.wrappers import Request, Response

def application(environ, start_response):
    response = Response('Hello Synergia!', mimetype='text/plain')
    return response(environ, start_response)

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 8000, application)

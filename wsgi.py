from app import app

if __name__ == '__main__':
    print('WSGI IS MAIN!')
    app.run(debug=False)
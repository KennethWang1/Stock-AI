import flask 
import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

app = flask.Flask(__name__)
api_key = os.getenv('POLYGONIO_API_KEY')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,ngrok-skip-browser-warning')    
    return response

@app.route('/api/v1/results', methods=['GET'])
def get_results():
    ticker = flask.request.args.get('ticker')
    if not ticker:
        return flask.jsonify({'error': 'Ticker parameter is required'}), 400
    
    try:
        root_dir = "./"
        directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        print(directories)
        if ticker in directories:
            file_path = f'{ticker}/today.json'
        
            with open(file_path, 'r') as file:
                data = json.load(file)
                r = requests.get(f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={api_key}")
                response = r.json()
                data['morningAnalysis']['currentPrice'] = response.get('p', 0)
                return flask.jsonify(data)
        return flask.jsonify({'error': f'No data found for ticker {ticker}'}), 404
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
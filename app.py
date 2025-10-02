import flask 
import json
import os

app = flask.Flask(__name__)

@app.route('/api/v1/results', methods=['GET'])
def get_results():
    ticker = flask.request.args.get('ticker')
    if not ticker:
        return flask.jsonify({'error': 'Ticker parameter is required'}), 400
    
    try:
        file_path = f'{ticker}/today.json'
        if not os.path.exists(file_path):
            return flask.jsonify({'error': 'Data not found'}), 404
            
        with open(file_path, 'r') as file:
            data = json.load(file)
            return flask.jsonify(data)
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
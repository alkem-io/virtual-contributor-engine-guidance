from flask import Flask, request
from flask_restful import Resource, Api

import ai_utils
import def_ingest
import openai
import json
from json import JSONEncoder

app = Flask(__name__)
api = Api(app)

class Chatbot:
    def __init__(self):
        self.chat_history = []
        self.docs = []
        self.chain = ai_utils.setup_chain()

    def reset_history(self):
        self.chat_history = []
        self.docs = []

    def submit_query(self, query):
        llm_result = self.chain(
            {"question": query, "chat_history": self.chat_history}
        )
        self.chat_history.append(
            (llm_result["question"], llm_result["answer"])
        )
        return {"question": llm_result["question"], "answer": llm_result["answer"], "sources":str(llm_result["source_documents"])}

    def get_chat_history(self):
        return self.chat_history

    def get_docs(self):
        return self.docs

class QueryAPI(Resource):
    def post(self):
        try:
            json_data = request.get_json(force=True)
            query = json_data['query']
            # Process the query and return the result
            print(query)
            result = alkemio_chatbot.submit_query(query)
            print (result)
            return {'result': result}, 200
        except KeyError:
            return {'error': 'No query key provided'}, 400

class ResetAPI(Resource):
    def get(self):
        # Here you should define what you want to do when /reset is called
        # For now, let's just return a simple message
        alkemio_chatbot.reset_history()
        return {'result': 'Reset performed successfully'}, 200

class IngestAPI(Resource):
    def get(self):
        # Here you should define what you want to do when /ingest is called
        # For now, let's just return a simple message
        def_ingest.mainapp()
        return {'result': 'Ingest performed successfully'}, 200

def_ingest.mainapp()

print("define chatbot")
alkemio_chatbot=Chatbot()

print("setup chain")
ai_utils.setup_chain()

api.add_resource(QueryAPI, '/query')
api.add_resource(ResetAPI, '/reset')
api.add_resource(IngestAPI, '/ingest')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
import flask
import jsonpickle
from flask import Flask, request, Response, jsonify
from src.utilities import andy_logger, andy_singleton, constants
from src.services import factes_engine

logger = andy_logger.get_logger("main")

andy_singleton.Singletons.get_instance()
tmp_org_id = None  # used to catch and reset the catch if new organization request is made
request_count = 0
app = Flask(__name__)


@app.route("/lq_detection", methods=["POST", "GET"])
def lq_detection():
    """
    This service is currently based on organization but later can be shifted to product

    Note:- input snippet must be list of json object with following parameters
    ["id", //requires
    "text",//required
    "speaker" ,// required
    "from_time",// required
    "to_time", // required
    "confidence "] //not-required but can be passed

    :return: snippets matched with lead_qualification facets and the total count of
    the matched facet_signals along each facet
    """
    global tmp_org_id, request_count

    org_id = request.args.get("org_id")
    snippet = request.json
    if org_id is None:
        org_id = constants.fetch_constant("default_organization")
    if request_count == 0:
        logger.info("This is the first request for organization={}".format(org_id))
        tmp_org_id = org_id
        request_count += 1
        matches = factes_engine.return_mached_facets(snippet, org_id)
        if len(matches) != 0:
            result = []
            for item in matches:
                tmp_dict = {"Snippet_id": item.snippet.sid, "Snippet_text": item.snippet_text,
                            "Snippet_Ques": item.snippet_question, "Facet_Name": item.facet_name,
                            "Facet_Signal_id": item.facet_signal.fsid, "Face_Signal": item.facet_signal_text,
                            "Score": str(item.score)}
                result.append(tmp_dict)
            resp = Response(jsonpickle.encode(result), mimetype='application/json')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            return str([])

    elif request != 0 and tmp_org_id != org_id:
        logger.info(
            "First request for new organization={} clearing the cache_facets for old_org={}".format(org_id, tmp_org_id))
        factes_engine.refresh_cached_lq_facets()
        request_count = 1
        tmp_org_id = org_id
        matches = factes_engine.return_mached_facets(snippet, org_id)
        if len(matches) != 0:
            result = []
            for item in matches:
                tmp_dict = {"Snippet_id": item.snippet.sid, "Snippet_text": item.snippet_text,
                            "Snippet_Ques": item.snippet_question, "Facet_Name": item.facet_name,
                            "Facet_Signal_id": item.facet_signal.fsid, "Face_Signal": item.facet_signal_text,
                            "Score": str(item.score)}
                result.append(tmp_dict)
            resp = Response(jsonpickle.encode(result), mimetype='application/json')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            return str([])
    else:
        request_count += 1
        logger.info("This is {} request for {} organization".format(request_count, tmp_org_id))
        matches = factes_engine.return_mached_facets(snippet, org_id)
        if len(matches) != 0:
            result = []
            for item in matches:
                tmp_dict = {"Snippet_id": item.snippet.sid, "Snippet_text": item.snippet_text,
                            "Snippet_Ques": item.snippet_question, "Facet_Name": item.facet_name,
                            "Facet_Signal_id": item.facet_signal.fsid, "Face_Signal": item.facet_signal_text,
                            "Score": str(item.score)}
                result.append(tmp_dict)
            resp = Response(jsonpickle.encode(result), mimetype='application/json')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            return str([])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="7777", debug=True)

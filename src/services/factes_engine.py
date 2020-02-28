from src.utilities import andy_singleton, andy_logger, constants
import pandas as pd
import os
from src.utilities.objects import Facet, FacetSignal, CaughtFacetSignals, VadChunk
import numpy as np
import time
from src.services import snippet_service
from src.utilities.andy_exceptions import NoFacetFound

logger = andy_logger.get_logger("facet_engine")


def refresh_cached_lq_facets():
    """
    This method refreshes the cached_facets singleton ,at present we only store facests for one organization and clear
    it if a new request is made with different organization id
    :return:
    """

    andy_singleton.Singletons.get_instance().get_cached_lq_facets().clear()
    logger.info("Refreshed lead_qualification facets cache")


def make_cached_lq_facets(org_id):
    """
    This method at present reads the facet signals from a file and creates the cached embeddings for the facet_signal
    on first request for the particular organization
    :param org_id:
    :return:
    """

    if len(andy_singleton.Singletons.get_instance().get_cached_lq_facets()) == 0:
        logger.info("Creating cached_facet_signals for organization={}".format(org_id))
        file_path = os.path.join(os.path.abspath(constants.fetch_constant("facet_folder_path")),
                                 constants.fetch_constant("facet_file_start_name") + str(org_id) + ".xlsx")
        if os.path.exists(file_path):
            logger.info("Getting data from file={}".format(file_path))
            facet_names = constants.fetch_constant("facet_names")
            df = pd.read_excel(file_path, encoding="utf-8")
            authority_facets = df[facet_names[0]].dropna().to_list()
            interest_facets = df[facet_names[1]].dropna().to_list()
            budget_facets = df[facet_names[2]].dropna().to_list()
            need_facets = df[facet_names[3]].dropna().to_list()
            logger.info("Making embeddings for the fetched facet_signals")
            start = time.time()
            all_facet_vectors = andy_singleton.Singletons.get_instance().perform_laser_embedding(
                authority_facets + budget_facets + interest_facets + need_facets)
            logger.info("Embedded {} sentences in {} secs".format(
                len(authority_facets + interest_facets + need_facets + budget_facets), time.time() - start))
            authority_facets_vectors = all_facet_vectors[:len(authority_facets)]
            budget_facets_vectors = all_facet_vectors[
                                    len(authority_facets):len(authority_facets) + len(budget_facets)]
            interest_facets_vectors = all_facet_vectors[
                                      len(authority_facets) + len(budget_facets):len(authority_facets) + len(
                                          budget_facets) + len(interest_facets)]
            need_facets_vectors = all_facet_vectors[len(authority_facets) + len(budget_facets) + len(interest_facets):]
            logger.info("Making facet_Signals")
            authority_facets_signals = []
            for i, item in enumerate(authority_facets_vectors):
                authority_facets_signals.append(
                    FacetSignal(i, authority_facets[i], [item], constants.fetch_constant("embedding_method")))
            interest_facets_signals = []
            for i, item in enumerate(interest_facets_vectors):
                interest_facets_signals.append(
                    FacetSignal(i, interest_facets[i], [item], constants.fetch_constant("embedding_method")))
            budget_facets_signals = []
            for i, item in enumerate(budget_facets_vectors):
                budget_facets_signals.append(
                    FacetSignal(i, budget_facets[i], [item], constants.fetch_constant("embedding_method")))
            need_facets_signals = []
            for i, item in enumerate(need_facets_vectors):
                need_facets_signals.append(
                    FacetSignal(i, need_facets[i], [item], constants.fetch_constant("embedding_method")))
            authority = Facet(0, facet_names[0], authority_facets_signals)
            intrest = Facet(1, facet_names[1], interest_facets_signals)
            budget = Facet(2, facet_names[2], budget_facets_signals)
            need = Facet(3, facet_names[3], need_facets_signals)
            andy_singleton.Singletons.get_instance().get_cached_lq_facets()[
                list(constants.fetch_constant("facet_names"))[0]] = authority
            andy_singleton.Singletons.get_instance().get_cached_lq_facets()[
                list(constants.fetch_constant("facet_names"))[1]] = intrest
            andy_singleton.Singletons.get_instance().get_cached_lq_facets()[
                list(constants.fetch_constant("facet_names"))[2]] = budget
            andy_singleton.Singletons.get_instance().get_cached_lq_facets()[
                list(constants.fetch_constant("facet_names"))[3]] = need
            logger.info("Cached facets for organization={}".format(org_id))

        else:
            logger.info("There are no facets defined for the {} organization".format(org_id))
            raise NoFacetFound(org_id, file_path)
    else:
        logger.info("Skipping caching of keyword for organization={}, they already exist in RAM".format(org_id))


def wrapper_method(snippets, org_id):
    try:
        make_cached_lq_facets(org_id)
        vad_chunk_list = []
        for item in snippets:
            vad_chunk_list.append(
                VadChunk(item["id"], item["from_time"], item["to_time"], item["speaker"], item["text"], None,
                         questions=None, q_encoding=None,
                         encoding_method=None))
        valid_vad_chunk_list = []
        for vad_chunk in vad_chunk_list:
            if snippet_service.check_snippet_speaker(vad_chunk):
                logger.info("Speaker = Agent for snippet_id={}".format(vad_chunk.sid))
                snippet_service.find_snippet_questions(vad_chunk)
                snippet_service.make_snippet_question_embeddings(vad_chunk)
                valid_vad_chunk_list.append(vad_chunk)
            else:
                logger.info("Speaker = Customer for snippet_id ={}".format(vad_chunk.sid))
        caught_facets = []
        for vad_chunk in valid_vad_chunk_list:
            if vad_chunk.q_encoding is not None:
                for i, question in enumerate(vad_chunk.questions):
                    for facet in andy_singleton.Singletons.get_instance().get_cached_lq_facets():
                        for facet_signal in andy_singleton.Singletons.get_instance().get_cached_lq_facets()[
                            facet].facet_signals:
                            score = (np.dot([vad_chunk.q_encoding[i]], np.array(facet_signal.embedding).T) / (
                                    np.linalg.norm([vad_chunk.q_encoding[i]]) * np.linalg.norm(
                                facet_signal.embedding)))[0][0]
                            if score >= constants.fetch_constant("threshold"):
                                caught_facets.append(
                                    CaughtFacetSignals(vad_chunk, vad_chunk.text, question, facet, facet_signal,
                                                       facet_signal.text, score,
                                                       constants.fetch_constant("embedding_method")))
        return make_result(caught_facets)

    except NoFacetFound as e:
        logger.error(e.message)
        pass
    return []


def make_result(caught_facets):
    if len(caught_facets) != 0:
        result = []
        a_id = []
        b_id = []
        i_id = []
        n_id = []
        for item in caught_facets:
            tmp_dict = {"Snippet_id": item.snippet.sid, "Snippet_text": item.snippet_text,
                        "Snippet_Ques": item.snippet_question, "Facet_Name": item.facet_name,
                        "Facet_Signal_id": item.facet_signal.fsid, "Face_Signal_text": item.facet_signal_text,
                        "Score": str(item.score)}
            result.append(tmp_dict)
            if item.facet_name.lower() == "authority":
                a_id.append(item.facet_signal.fsid)
            elif item.facet_name.lower() == "interest":
                i_id.append(item.facet_signal.fsid)
            elif item.facet_name.lower() == "budget":
                b_id.append(item.facet_signal.fsid)
            else:
                n_id.append(item.facet_signal.fsid)
        count_dict = [{"Facet": "authority", "count": len(set(a_id)), "facetsignal_id": set(a_id)},
                      {"Facet": "interest", "count": len(set(i_id)), "facetsignal_id": set(i_id)},
                      {"Facet": "budget", "count": len(set(b_id)), "facetsignal_id": set(b_id)},
                      {"Facet": "need_investigation", "count": len(set(n_id)), "facetsignal_id": set(n_id)}]
        return result, count_dict
    else:
        return []


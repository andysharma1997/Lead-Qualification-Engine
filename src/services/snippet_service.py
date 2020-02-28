from src.services import question_detection
from src.utilities import andy_singleton, andy_logger, constants

logger = andy_logger.get_logger("snippet_service")


def check_snippet_speaker(vad_chunk):
    """
    This method checks if the snippet speaker is agent or customer
    :param vad_chunk:
    :return: true if speaker is agent else false
    """
    logger.info("Checking speaker")
    return True if vad_chunk.speaker == "Agent" else False


def find_snippet_questions(vad_chunk):
    """
    Extractes questions from snippet text and sets the snippet question if any question is detected else sets if to None
    :param vad_chunk:
    :return: None
    """
    questions = question_detection.find_questions(vad_chunk.text)
    if len(questions) != 0:
        logger.info("Found {} question for snippet_id={}".format(len(questions), vad_chunk.sid))
        vad_chunk.set_questions(questions)
    else:
        logger.info("Did not find any question in snippet {}".format(vad_chunk.sid))
        vad_chunk.set_questions(None)


def make_snippet_question_embeddings(vad_chunk):
    """
    Sets the sentence embedding of snippet questions if present else sets it to None
    :param vad_chunk:
    :return: None
    """
    if vad_chunk.questions is not None:
        vad_chunk.set_question_encoding(
            andy_singleton.Singletons.get_instance().perform_laser_embedding(vad_chunk.questions),
            constants.fetch_constant("embedding_method"))
        logger.info(
            "Calculated  embeddings for {} snippet questions for snippet_id ={}".format(
                len(vad_chunk.questions),
                vad_chunk.sid))
    else:
        logger.info("There were not snippet questions for snippet_id=".format(vad_chunk.sid))
        vad_chunk.set_question_encoding(None, None)

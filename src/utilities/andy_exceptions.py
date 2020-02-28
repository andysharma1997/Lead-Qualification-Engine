class AndyException(Exception):
    pass


class NoFacetFound(AndyException):
    """This exception is raised if thee are no facets for the particular organisation """

    def __init__(self, org_id, file_path):
        self.code = 9999
        self.message = "{}: The facet for organization={} were not found please check {} file_path".format(self.code,
                                                                                                           org_id,
                                                                                                           file_path)

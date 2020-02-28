import yaml
import os.path as path


def fetch_constant(constant_name, file_name="constants.yaml"):
    p = path.abspath(path.join(__file__, "../../resources/" + str(file_name)))
    with open(p, 'r') as f:
        doc = yaml.safe_load(f)
    return doc[constant_name]

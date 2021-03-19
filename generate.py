"""
Generate a dataset to evaluate bias in sense embeddings
"""

def people_template():
    """
    Generate nationalities vs. languages sense related stereotypes
    """
    nationalities = ["Japanese", "Chinese", "English", "Arabic", "German",
                     "French", "Spanish", "Portuguese", "Norwegian", "Swedish", "Polish", "Romanian",
                     "Russian", "Egyptian", "Finnish", "Vietnamese"]
    people_pleasant = ["beautiful", "nice", "friendly", "kind", "good looking", 
                    "intelligent", "clever", "generous", "funny", "cute", "handsome", "innocent"]
    people_unpleasant = ["ugly", "arrogant", "unkind", "unfriendly", "uppish", "foolish", "stupid", "cunning", "silly"]

    language_pleasant = ["easy to learn", "beautiful", "elegant", "soft", "easy to understand", "easy to write"]
    language_unpleasant = ["difficult to learn", "ugly", "rough", "hash", "difficult to understand", "difficult to write"]

    templates = [("$1 people are $2.\t[nationality, $3]", nationalities, people_pleasant, "anti"), 
                ("$1 people are $2.\t[nationality, $3]", nationalities, people_unpleasant, "stereo"),
                ("$1 people are $2.\t[nationality, $3]", ["Black"], people_pleasant, "anti"), 
                ("$1 people are $2.\t[nationality, $3]", ["Black"], people_unpleasant, "stereo"),
                ("$1 language is $2.\t[language, $3]", nationalities, language_pleasant, "anti"),
                ("$1 language is $2.\t[language, $3]", nationalities, language_unpleasant, "stereo")]
    inst = []
    for template in templates:
        pattern = template[0]
        targets = template[1]
        attributes = template[2]
        label = template[3]
        for target in targets:
            for attribute in attributes:
                txt = pattern.replace("$1", target)
                txt = txt.replace("$2", attribute)
                txt = txt.replace("$3", label)
                inst.append(txt)
    return inst

def write_to_file(instances, fname):
    """
    Write the instances to a file.
    """
    with open(fname, 'w') as F:
        for inst in instances:
            F.write("%s\n" % inst)
    pass



if __name__ == "__main__":
    instances = people_template()
    write_to_file(instances, "racial-bias.txt")

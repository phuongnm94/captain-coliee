import xml.etree.ElementTree as Et
import glob

def format_first_line(text):
    lines = text.split("\n")
    results = []
    for line in lines:
        if line[0] == "":
            continue
        if line[0] == "(" and line[-1] == ")":
            continue
        results.append(line)
    return "\n".join(results)

def load_samples(filexml):
    # try:
        tree = Et.parse(filexml)
        root = tree.getroot()
        samples = []
        for i in range(0, len(root)):
            sample = {'result': []}
            for j, e in enumerate(root[i]):
                if e.tag == "t1":
                    sample['result'] = format_first_line(e.text.strip())
                elif e.tag == "t2":
                    question = e.text.strip()
                    sample['content'] = question if len(question) > 0 else None
            sample.update(
                {'index': root[i].attrib['id'], 'label': root[i].attrib.get('label', "N")})

            # filter the noise samples
            if sample['content'] is not None:
                samples.append(sample)
            else:
                print("[Important warning] samples {} is ignored".format(sample))
        return samples

def load_test_data_samples(path_folder_base, test_id):
    data = []
    test = load_samples(f"{path_folder_base}/riteval_{test_id}.xml")
    for file_path in glob.glob(f"{path_folder_base}/riteval_{test_id}.xml"):
        data = data + load_samples(file_path)
    return data


def load_all_data_samples(path_folder_base):
    data = []
    for file_path in glob.glob("{}/*.xml".format(path_folder_base)):
        data = data + load_samples(file_path)
    return data

# data = load_samples("data/en/riteval_R04_en.xml")
# print(data[0])
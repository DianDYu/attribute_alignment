import sys
import json
import random


def main():
    extracted_data = []

    # bst data: 4819 conversations
    with open("extracted_bst.json") as bst_file:
        bst_data = json.load(bst_file)
        for ori_conversation in bst_data:
            conversation = {"dialog": []}
            ori_dialog = ori_conversation["dialog"]
            conversation["dialog"].append(ori_conversation["context"][0])
            for turn in ori_dialog:
                conversation["dialog"].append(turn["text"])
            extracted_data.append(conversation)

    # wow data: 18430 conversations
    with open("extracted_wow_train.json") as wow_file:
        wow_data = json.load(wow_file)
        for ori_conversation in wow_data:
            conversation = {"dialog": []}
            ori_dialog = ori_conversation["dialog"]
            for i, turn in enumerate(ori_dialog):
                # fix "let's talk about topic" if apprentice starts the conversation
                if i == 0 and turn["text"].startswith("let's talk about"):
                    continue
                conversation["dialog"].append(turn["text"])
            extracted_data.append(conversation)

    # convai data: 18193 conversations
    with open("extracted_convai.json") as convai_file:
        convai_data = json.load(convai_file)
        for ori_conversation in convai_data:
            conversation = {"dialog": []}
            conversation["dialog"] = ori_conversation["dialog"]
            extracted_data.append(conversation)

    # ed data:  17779 conversations (was 17843, removed bad ones)
    with open("extracted_ed.json") as ed_file:
        ed_data = json.load(ed_file)
        for ori_conversation in ed_data:
            conversation = {"dialog": []}
            conversation["dialog"] = ori_conversation["dialog"]
            extracted_data.append(conversation)

    random.shuffle(extracted_data)

    # print(extracted_data[:2])

    with open("extracted_all.json", "w") as outfile:
        json.dump(extracted_data, outfile)


if __name__ == '__main__':
    main()
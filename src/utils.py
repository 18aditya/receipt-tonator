import json
import re

def json2token(obj, update_special_tokens_for_json_key, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        if sort_json_key:
            keys = sorted(obj.keys(), reverse=True)
        else:
            keys = obj.keys()
            
        for k in keys:
            update_special_tokens_for_json_key(k)
            output += (
                fr"<s_{k}>"
                + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                + fr"</s_{k}>"
            )
        return output
    elif type(obj) == list:
        return r"".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        obj = str(obj)
        if f"<{obj}/>" in ["<sep/>", "<unk/>"]: # exclude special tokens
            return obj
        return obj

def token2json(tokens, is_inner_value=False, added_vocab=None):
    """
    Convert a token sequence back into a JSON object (for inference/validation)
    """
    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        start_token_match = start_token.group()
        end_token_match = f"</s_{key}>"
        
        end_token = re.search(end_token_match, tokens, re.IGNORECASE)
        if end_token is None:
            tokens = tokens.replace(start_token_match, "")
            continue
            
        start_token_index = start_token.start()
        end_token_index = end_token.start()
        
        value = tokens[start_token_index + len(start_token_match) : end_token_index]
        tokens = tokens[end_token_index + len(end_token_match) :]
        
        if key in output:
            if not isinstance(output[key], list):
                output[key] = [output[key]]
            output[key].append(token2json(value, is_inner_value=True, added_vocab=added_vocab) if "<s_" in value else value)
        else:
            output[key] = token2json(value, is_inner_value=True, added_vocab=added_vocab) if "<s_" in value else value

    return output if output else tokens

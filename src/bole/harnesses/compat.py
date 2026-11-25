"""Compatibility transforms enabled only by configuration."""
import copy
import json

def normalize_request(body, protocol):
    body = copy.deepcopy(body)
    changes = []
    if protocol == "responses":
        inputs, tools = [], {}
        for tool in body.get("tools", []):
            tools[(tool["type"], tool.get("name"))] = tool
        for item in body.get("input", []):
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                for tool in item.get("tools", []):
                    tools[(tool["type"], tool.get("name"))] = tool
                changes.append("additional_tools_to_top_level")
            else:
                inputs.append(item)
                if (
                    isinstance(item, dict)
                    and item.get("type") in {"custom_tool_call_output", "function_call_output"}
                    and isinstance(item.get("output"), list)
                ):
                    images = [b for b in item["output"] if b.get("type") == "input_image"]
                    if images:
                        item["output"] = [
                            b for b in item["output"] if b.get("type") != "input_image"
                        ]
                        inputs.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "Images returned by call_id=" + item["call_id"],
                                    },
                                    *images,
                                ],
                            }
                        )
                        changes.append("tool_output_images_to_user_message")
        if changes:
            body["input"] = inputs
            body["tools"] = list(tools.values())
    elif protocol == "messages":
        for message in body.get("messages", []):
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            content = []
            for block in message["content"]:
                content.append(block)
                if block.get("type") != "tool_result" or not isinstance(block.get("content"), list):
                    continue
                images = [b for b in block["content"] if b.get("type") == "image"]
                if images:
                    block["content"] = [b for b in block["content"] if b.get("type") != "image"]
                    content.append(
                        {
                            "type": "text",
                            "text": "Images returned by tool_use_id=" + block["tool_use_id"],
                        }
                    )
                    content.extend(images)
                    changes.append("tool_result_images_to_same_user_message")
            message["content"] = content
    return body, changes


def stream_complete(body):
    """Recognize protocol termination separately from a clean HTTP connection close."""
    data = []
    for line in [*body.decode(errors="replace").splitlines(), ""]:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                return True
            try:
                value = json.loads(payload)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("type") in {
                "message_stop",
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                return True
    return False

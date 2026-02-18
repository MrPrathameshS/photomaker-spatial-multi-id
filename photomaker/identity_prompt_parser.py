import torch


def extract_identity_prompt_map_clean(
    text_input_ids,
    identity_token_indices,
):
    """
    Build identity → prompt token routing
    based on identity token positions.
    """

    identity_prompt_map = {}
    seq_len = len(text_input_ids)

    for i, token_positions in enumerate(identity_token_indices):

        if len(token_positions) == 0:
            continue

        # Start routing AFTER identity tokens
        start = max(token_positions) + 1

        # End before next identity
        if i < len(identity_token_indices) - 1:
            next_identity_start = min(identity_token_indices[i + 1])
            end = next_identity_start
        else:
            end = seq_len

        identity_prompt_map[i] = list(range(start, end))

    return identity_prompt_map

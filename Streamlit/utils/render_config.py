import streamlit as st


def render_config_from_schema(schema: dict) -> dict:
    config = {}
    props = schema["properties"]

    for name, meta in props.items():
        if name == "model_type":
            config[name] = meta.get("default", "unknown")
            continue

        label = meta.get("title", name).replace("_", " ").capitalize()
        help_text = meta.get("description", "")
        default = meta.get("default")

        if meta.get("type") == "integer":
            config[name] = st.number_input(label, value=default or 0, help=help_text)
        elif meta.get("type") == "number":
            config[name] = st.number_input(label, value=default or 0.0, format="%.4f", help=help_text)
        elif meta.get("type") == "boolean":
            config[name] = st.checkbox(label, value=default or False, help=help_text)
        elif meta.get("type") == "string" and "enum" in meta:
            index = meta["enum"].index(default) if default else 0
            config[name] = st.selectbox(label, meta["enum"], index=index, help=help_text)
        else:
            config[name] = st.text_input(label, value=str(default or ""), help=help_text)

    return config

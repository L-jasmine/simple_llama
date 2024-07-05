pub fn llama3_prompt_template() -> super::LlmPromptTemplate {
    fn encode_string(content: &[super::Content]) -> String {
        let mut result = String::new();
        for c in content.iter() {
            result.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>{}<|eot_id|>",
                c.role, c.message
            ));
        }
        result
    }

    fn is_end_of_header(token: &str) -> bool {
        token == "<|end_header_id|>"
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle: super::LlmPromptTemplate::identity,
    }
}

pub fn hermes_2_pro_llama3_prompt_template() -> super::LlmPromptTemplate {
    fn encode_string(content: &[super::Content]) -> String {
        let mut result = String::new();
        for c in content.iter() {
            result.push_str(&format!(
                "<|im_start|>{}\r\n{}<|im_end|>\r\n",
                c.role, c.message
            ));
        }
        result
    }

    fn is_end_of_header(_token: &str) -> bool {
        true
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle: super::LlmPromptTemplate::identity,
    }
}

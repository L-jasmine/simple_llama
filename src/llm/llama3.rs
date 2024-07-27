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

    fn post_handle(token: String, content: &str) -> Option<String> {
        if token == "<|eot_id|>" || content.ends_with("<|eot_id|>") || content.ends_with("#") {
            None
        } else {
            Some(token)
        }
    }

    fn post_handle_content(content: &mut String) {
        if content.ends_with("<|eot_id|>") {
            let bs = unsafe { content.as_mut_vec() };
            let len = bs.len();
            bs.truncate(len - "<|eot_id|>".len());
        }
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle,
        post_handle_content,
    }
}

pub fn hermes_2_pro_llama3_prompt_template() -> super::LlmPromptTemplate {
    fn encode_string(content: &[super::Content]) -> String {
        let mut result = String::new();
        for c in content.iter() {
            result.push_str(&format!(
                "<|im_start|>{}\r{}<|im_end|>\r",
                c.role, c.message
            ));
        }
        result
    }

    fn is_end_of_header(_token: &str) -> bool {
        true
    }

    fn post_handle_content(content: &mut String) {
        if content.ends_with("<|im_end|>") {
            let bs = unsafe { content.as_mut_vec() };
            let len = bs.len();
            bs.truncate(len - "<|im_end|>".len());
        }
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle: super::LlmPromptTemplate::identity,
        post_handle_content,
    }
}

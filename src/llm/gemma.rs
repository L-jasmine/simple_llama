pub fn gemma2_prompt_template() -> super::LlmPromptTemplate {
    fn encode_string(content: &[super::Content]) -> String {
        let mut result = String::new();
        for c in content.iter() {
            result.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>",
                c.role, c.message
            ));
        }
        result
    }

    fn is_end_of_header(token: &str) -> bool {
        token.ends_with("\n")
    }

    fn post_handle(token: String, content: &str) -> Option<String> {
        if token == "<end_of_turn>" || content.ends_with("<end_of_turn>") {
            None
        } else {
            Some(token)
        }
    }

    fn post_handle_content(content: &mut String) {
        if content.ends_with("<end_of_turn>") {
            let bs = unsafe { content.as_mut_vec() };
            let len = bs.len();
            bs.truncate(len - b"<end_of_turn>".len());
        }
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle,
        post_handle_content,
    }
}

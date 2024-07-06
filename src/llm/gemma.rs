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

    fn post_handle(token: String) -> Option<String> {
        if token == "<end_of_turn>" {
            None
        } else {
            Some(token)
        }
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle,
    }
}

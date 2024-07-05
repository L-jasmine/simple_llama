use crate::llm::LlmPromptTemplate;

pub fn qwen_prompt_template() -> super::LlmPromptTemplate {
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

    fn is_end_of_header(token: &str) -> bool {
        token.ends_with("\n")
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle: LlmPromptTemplate::identity,
    }
}

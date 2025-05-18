import axios from "axios";

const defaultBaseUrl = "http://localhost:11434";

export interface ChatMessage {
	roll: "system" | "user" | "assistant" | "tool";
	content: string;
}

export interface ChatCompletionRequest {
	model: string;
	messages: ChatMessage[] | undefined;
}

export interface ChatResponse {
	model: string;
	created_at: string;
	message: ChatMessage;
	done: boolean;
}

/**
 * Ollama API の /api/chat エンドポイントを呼び出す
 * @param req リクエストボディ
 * @param baseUrl APIサーバーのベースURL（デフォルト: "http://localhost:11434"）
 * @throws APIエラー時はErrorをスロー
 */
export async function generateChatCompletion(
	req: ChatCompletionRequest,
	baseUrl = defaultBaseUrl,
): Promise<ChatResponse> {
	const response = await axios.post(`${baseUrl}/api/chat`, req, {
		headers: {
			"Content-Type": "application/json",
		},
	});
	return response.data;
}

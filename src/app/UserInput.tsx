"use client";

import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import ollama from "ollama/browser";

function AssistantSelect() {
	return (
		<TextField
			label="システムプロンプト"
			variant="outlined"
			fullWidth
			multiline
		/>
	);
}

export default function UserInput() {
	const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
		if (event.ctrlKey && event.key === "Enter") {
			event.preventDefault();
			// メッセージを送信する
			(async () => {
				console.log("request chat");
				const response = await ollama.chat({
					model: "gemma3:4b",
					messages: [{ role: "user", content: "Why is the sky blue?" }],
				});
				console.log(response);
			})();
		}
	};

	return (
		<Stack spacing={1}>
			<TextField
				fullWidth
				label="新しいメッセージ (Ctrl+Enter)"
				variant="outlined"
				multiline
				onKeyDown={handleKeyDown}
			/>
			<AssistantSelect />
		</Stack>
	);
}

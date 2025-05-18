"use client";

import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";

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
	const handleCtrlEnter = (event: React.KeyboardEvent<HTMLInputElement>) => {
		if (event.ctrlKey && event.key === "Enter") {
			event.preventDefault();
			// TODO: メッセージ送信処理をここに実装
			console.log("Ctrl+Enter pressed");
		}
	};

	return (
		<Stack spacing={1}>
			<TextField
				fullWidth
				label="新しいメッセージ (Ctrl+Enter)"
				variant="outlined"
				multiline
				onKeyDown={handleCtrlEnter}
			/>
			<AssistantSelect />
		</Stack>
	);
}

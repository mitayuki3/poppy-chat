import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";

export default function Home() {
	return (
		<main>
			<Stack spacing={1}>
				<TextField
					label="システムプロンプト"
					variant="outlined"
					fullWidth
					multiline
				/>
				<TextField multiline defaultValue="Hello world!" />
				<TextField
					fullWidth
					label="新しいメッセージ (Ctrl+Enter)"
					variant="outlined"
				/>
			</Stack>
		</main>
	);
}

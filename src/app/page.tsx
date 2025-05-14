import TextField from "@mui/material/TextField";

export default function Home() {
	return (
		<main>
			<TextField
				label="システムプロンプト"
				variant="outlined"
				fullWidth
				margin="normal"
				multiline
				minRows={3}
			/>
			<div>Hello world!</div>
		</main>
	);
}

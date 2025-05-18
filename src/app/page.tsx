import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";

import UserInput from "./UserInput";

export default function Home() {
	return (
		<main>
			<Stack spacing={1}>
				<TextField multiline defaultValue="Hello world!" />
				<UserInput />
			</Stack>
		</main>
	);
}

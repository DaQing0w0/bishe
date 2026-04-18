import argparse
import pandas as pd

DEFAULT_INPUT_CSV = "/root/workspace/bishe/mgpusim-4.1.1/amd/samples/minerva/mem.csv"
DEFAULT_OUTPUT_CSV = "/root/workspace/bishe/epoch0_first_n_minus_m.csv"


def count_epoch_0_and_1(file_path, chunk_size=500000):
	n_epoch0 = 0
	m_epoch1 = 0

	reader = pd.read_csv(file_path, usecols=["epoch"], chunksize=chunk_size)
	for chunk in reader:
		epoch_series = pd.to_numeric(chunk["epoch"], errors="coerce")
		n_epoch0 += int((epoch_series == 0).sum())
		m_epoch1 += int((epoch_series == 1).sum())

	return n_epoch0, m_epoch1


def write_epoch0_first_n_minus_m_rows(file_path, output_path, chunk_size=500000):
	n_epoch0, m_epoch1 = count_epoch_0_and_1(file_path, chunk_size)
	rows_to_take = n_epoch0 - m_epoch1

	print(f"n (epoch==0): {n_epoch0}")
	print(f"m (epoch==1): {m_epoch1}")
	print(f"n - m: {rows_to_take}")

	if rows_to_take <= 0:
		print("n - m <= 0, no rows to output.")
		# Create an empty output file with headers.
		sample = pd.read_csv(file_path, nrows=0)
		sample.to_csv(output_path, index=False)
		return

	remaining = rows_to_take
	header_written = False

	reader = pd.read_csv(file_path, chunksize=chunk_size)
	for chunk in reader:
		if remaining <= 0:
			break

		epoch_series = pd.to_numeric(chunk["epoch"], errors="coerce")
		part = chunk[epoch_series == 0].head(remaining)
		if part.empty:
			continue

		part.to_csv(
			output_path,
			mode="a" if header_written else "w",
			index=False,
			header=not header_written,
		)

		header_written = True
		remaining -= len(part)

	written_rows = rows_to_take - remaining
	print(f"written rows: {written_rows}")
	print(f"output file: {output_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Output first n-m rows with epoch==0, where n=count(epoch==0) and m=count(epoch==1)."
	)
	parser.add_argument(
		"--input",
		default=DEFAULT_INPUT_CSV,
		help="Path to input CSV file",
	)
	parser.add_argument(
		"--output",
		default=DEFAULT_OUTPUT_CSV,
		help="Path to output CSV file",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=500000,
		help="Chunk size for reading CSV",
	)
	args = parser.parse_args()

	write_epoch0_first_n_minus_m_rows(
		file_path=args.input,
		output_path=args.output,
		chunk_size=args.chunk_size,
	)


if __name__ == "__main__":
	main()

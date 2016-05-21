defmodule Dataset.Mapper do
  def process(dataset_file, output_dir, subjects_file) do

    File.stream!(subjects_file)
    |> Stream.map(&String.strip/1)
    |> Enum.map(&(Task.async(Dataset.Mapper, :process_subject, [dataset_file, output_dir, &1])))
    |> Enum.map(&Task.await(&1, 1000000))
  end

  def process_subject(dataset_file, output_dir, subject) do
    {:ok, out} = File.open(output_dir <> "/" <> subject <> ".txt", [:write])
    File.stream!(dataset_file)
    |> Stream.drop(1)
    |> Stream.map(&String.strip/1)
    |> Stream.map(&(String.split(&1, ",")))
    |> Stream.map(&(map_line(&1, subject)))
    |> Stream.map(&create_out_line/1)
    |> Enum.each(&IO.binwrite(out, &1))

    # File.close(out)
  end

  def map_line([subject, _, _, _ | values], current_subject) do
    is_current = if subject == current_subject, do: 1, else: 0
    mapped_values = values
    |> Enum.map(&Float.parse/1)
    |> Enum.map(&(elem(&1, 0)))
    |> Enum.map(&(&1 * 1000))
    # |> Enum.take_every(3)

    [is_current | mapped_values]
  end

  def create_out_line(values) do
    out = values
    |> Enum.join(",")

    out <> "\n"
  end
end

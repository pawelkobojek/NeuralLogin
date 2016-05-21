defmodule Dataset.CLI do

  def main(argv) do
    argv
    |> parse_args
    |> process
  end

  def parse_args(argv) do
    parse = OptionParser.parse(argv, switches: [help: :boolean],
                                     aliases: [h: :help])

    case parse do
      {[help: true], _, _} -> :help
      {_, [dataset_file, output_dir, subjects_file], _} -> {dataset_file, output_dir, subjects_file}
      _ -> :help
      # _ -> {Application.get_env(:dataset, :dataset_file), Application.get_env(:dataset, :output_dir),
                  # Application.get_env(:dataset, :subjects_file)}
    end
  end

  def process(:help) do
    IO.puts """
    USAGE: dataset <dataset_file> <output_dir> <subjects_file>
    """
    System.halt(0)
  end

  def process({dataset_file, output_dir, subjects_file}) do
    Dataset.Mapper.process(dataset_file, output_dir, subjects_file)
  end
end

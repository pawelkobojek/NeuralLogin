defmodule Features do

  def compute_features(mail, output_dir \\ ".", data_dir \\ ".") do
    {:ok, files} = File.ls(data_dir)
    extracted_lines = process_files(mail, files)
    {:ok, out} = File.open(output_dir <> "/" <> mail <> ".txt", [:write])
    extracted_lines |> Enum.each(fn line -> IO.binwrite(out, line) end)
  end

  def process_all_raw_data(data_dir \\ ".", output_dir \\ "../../training_data") do
    File.stream!("../emails.txt")
      |> Enum.map(&String.rstrip(&1))
      |> Enum.each(fn f -> Features.compute_features(f, output_dir, data_dir) end)
  end

  def process_files(_, []) do
    []
  end

  def process_files(mail, [file|t]) do
    [extract_features(mail, file) | process_files(mail, t)]
  end

  defp extract_features(mail, file) do
    import Stream

    {:ok, content} = File.read(file)
    [userMail|_] = content |> String.split("\n")
    label = if String.contains?(userMail, mail), do: 1, else: 0

    File.stream!(file)
      |> map(&String.split(&1, ";"))
      |> filter(fn [h|_] -> h == "keydown" end)
      |> filter(fn line -> length(line) >= 5 end)
      |> map(&List.to_tuple(&1))
      |> filter(fn line -> String.contains?(elem(line, 3), mail) or (label == 1 and String.contains?(elem(line, 3), "loginMail")) end)
      |> filter(fn line -> elem(line, 1) != "9" and elem(line, 1) != "13" end)
      |> map(&only_relevant(&1))
      |> Enum.group_by(fn line -> elem(line, 1) end)
      |> map(&parse_ints_and_label_data(&1, label))
      |> map(&deep_sort(&1))
      |> map(&compute_differences(&1))
      |> filter(fn line -> length(elem(line, 1)) > String.length(mail) end)
      |> Enum.map(fn line -> (line |> elem(0) |> to_string) <> "," <> (line |> elem(1) |> Enum.join(",")) <> "\n" end)
  end

  defp only_relevant(r) do
    {elem(r, 2), elem(r, 3)}
  end

  defp parse_ints_and_label_data(d, label) do
    {label, d |> elem(1) |> Enum.map(fn line ->
      {a, _} = Integer.parse(elem(line, 0))
      a
    end)}
  end

  defp deep_sort({key, value}) do
    {key, value |> Enum.sort}
  end

  defp compute_differences({key, value}) do
    {key, value |> map_two(&(&1 - &2))}
  end

  def map_two(list, fun) when is_list(list) do
    [h|_] = list
    map_two(list, fun, h)
  end

  def map_two([], _, _) do
  end

  def map_two([h|t], fun, acc) when t != [] do
    [fun.(h, acc)|map_two(t, fun, h)]
  end

  def map_two([h|_], fun, acc) do
    [fun.(h, acc)]
  end
end

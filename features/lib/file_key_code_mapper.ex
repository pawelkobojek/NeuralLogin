defmodule FileKeyCodeMapper do

  def to_char(filename) do
    output_file_name = filename <> ".output.txt"
    {:ok, output} = File.open(output_file_name, [:write])
    File.stream!(filename)
      |> Enum.map(&(String.split(&1, ";")))
      |> Enum.filter(fn [h|_] -> h == "keydown" end)
      |> Enum.map(&(replace(&1)))
      |> Enum.map(&(Enum.join(&1, ";")))
      |> Enum.each(&(IO.binwrite(output, &1)))

      File.close(output)
      output_file_name
  end

  def replace(list) when length(list) != 5 do
    list
  end

  def replace([h|_]) when h == "focusChanged" do
    ["\n----------------------\n\n"]
  end

  def replace(list) do
    replace(list, 2)
  end

  def replace([h|t], 1) do
    [map_to_key(h)|t]
  end

  def replace([h|t], i) do
    [h | replace(t, i-1)]
  end

  defp map_to_key(key_code) do
    case key_code do
      "8"   -> "backspace"
      "9"   -> "tab"
      "13"  -> "enter"
      "16"  -> "shift"
      "17"  -> "ctrl"
      "18"  -> "alt"
      "19"  -> "pause"
      "20"  -> "caps"
      "27"  -> "esc"
      "33"  -> "pgup"
      "34"  -> "pgdwn"
      "35"  -> "end"
      "36"  -> "home"
      "37"  -> "left"
      "38"  -> "up"
      "39"  -> "right"
      "40"  -> "down"
      "45"  -> "ins"
      "46"  -> "del"
      "48"  -> "0"
      "49"  -> "1"
      "50"  -> "2"
      "51"  -> "3"
      "52"  -> "4"
      "53"  -> "5"
      "54"  -> "6"
      "55"  -> "7"
      "56"  -> "8"
      "57"  -> "9"
      "65"  -> "a"
      "66"  -> "b"
      "67"  -> "c"
      "68"  -> "d"
      "69"  -> "e"
      "70"  -> "f"
      "71"  -> "g"
      "72"  -> "h"
      "73"  -> "i"
      "74"  -> "j"
      "75"  -> "k"
      "76"  -> "l"
      "77"  -> "m"
      "78"  -> "n"
      "79"  -> "o"
      "80"  -> "p"
      "81"  -> "q"
      "82"  -> "r"
      "83"  -> "s"
      "84"  -> "t"
      "85"  -> "u"
      "86"  -> "v"
      "87"  -> "w"
      "88"  -> "x"
      "89"  -> "y"
      "90"  -> "z"
      "91"  -> "lwin"
      "92"  -> "rwin"
      "93"  -> "select"
      "96"  -> "num0"
      "97"  -> "num1"
      "98"  -> "num2"
      "99"  -> "num3"
      "100"  -> "num4"
      "101"  -> "num5"
      "102"  -> "num6"
      "103"  -> "num7"
      "104"  -> "num8"
      "105"  -> "num9"
      "106"  -> "mul"
      "107"  -> "add"
      "109"  -> "sub"
      "110"  -> "decpt"
      "111"  -> "div"
      "112"  -> "f1"
      "113"  -> "f2"
      "114"  -> "f3"
      "115"  -> "f4"
      "116"  -> "f5"
      "117"  -> "f6"
      "118"  -> "f7"
      "119"  -> "f8"
      "120"  -> "f9"
      "121"  -> "f10"
      "122"  -> "f11"
      "123"  -> "f12"
      "144"  -> "nlock"
      "145"  -> "slock"
      "186"  -> ";"
      "187"  -> "="
      "188"  -> ","
      "189"  -> "-"
      "190"  -> "."
      "191"  -> "/"
      "192"  -> "`"
      "219"  -> "("
      "220"  -> "\\"
      "221"  -> ")"
      "222"  -> "'"
      _      -> "wtf"
    end
  end
end

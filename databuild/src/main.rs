// typstifier
// databuild/src/main.rs

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{anychar, char, multispace0, none_of, not_line_ending, one_of};
use nom::combinator::{map, value};
use nom::multi::{many0, many_till};
use nom::sequence::{delimited, preceded, terminated, tuple};
use nom::IResult;
use rand::Rng;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        println!("usage: databuild [filename] [datadir] [symboldir] [jsonfile]");
        return;
    }

    let filename = Path::new(&args[1]);
    let data_dir = Path::new(&args[2]);
    let symbol_dir = Path::new(&args[3]);
    let json_file = Path::new(&args[4]);

    let input = std::fs::read_to_string(filename).expect("Unable to read file.");
    let mut data: Vec<_> = generate_data(&input)
        .into_iter()
        .map(|d| (d, true))
        .collect();

    let mut rng = rand::thread_rng();
    let repetitions = 50;
    let total_size = 64;

    println!(
        "Todo: {} symbols (including whitespaces), {} repetitions, {} total",
        data.len(),
        repetitions,
        data.len() * repetitions
    );

    let mut index: usize = 0;
    'mainloop: for (d, b) in &mut data {
        let png_in = Path::new(symbol_dir).join(&format!("{index}.png"));
        gen_typst_image(d, &png_in);

        for i in 0..repetitions {
            let png_out = Path::new(data_dir).join(&format!("{index}-{i}.png"));
            let size = rng.gen_range(32..=64);

            let success = blur_and_place(
                &png_in,
                &png_out,
                total_size,
                size,
                rng.gen_range((-(total_size / 2) + size / 2)..=(total_size / 2 - size / 2)),
                rng.gen_range((-(total_size / 2) + size / 2)..=(total_size / 2 - size / 2)),
                rng.gen_range(-30..=30),
                rng.gen_range(0..=3),
                rng.gen_range(0..=3),
            );

            // If image is all white (whitespace characters) go to next one and don't increment index
            if !success {
                assert!(i == 0);
                std::fs::remove_file(&png_in).expect("Could not delete png file.");
                std::fs::remove_file(&png_out).expect("Could not delete png file.");
                *b = false;
                continue 'mainloop;
            }
        }
        blur_and_place(&png_in, &png_in, 128, 128, 0, 0, 0, 0, 0);
        index += 1;
    }

    let data = data
        .into_iter()
        .filter_map(|(d, b)| if b { Some(d) } else { None })
        .collect();
    output_json(&data, &json_file.to_path_buf());

    println!(
        "Done: {} symbols (without whitespaces), {} repetitions, {} total",
        data.len(),
        repetitions,
        data.len() * repetitions
    );
}

// Generate images

fn gen_typst_image(unicode: &TypstUnicode, png_filename: &PathBuf) {
    let temp_dir = std::env::temp_dir();
    let typst_doc = format!(
        "#set page(width: auto, height: auto, margin: (x: 8pt, y: 8pt))\n#set align(center + horizon)\n${}$\n",
        unicode.sym_name);
    let typst_filename = temp_dir.join("typstifier.typ");

    std::fs::write(&typst_filename, &typst_doc).expect("Unable to write typst file.");

    let status = Command::new("typst")
        .arg("c")
        .arg("--ppi")
        .arg("1440")
        .arg(&typst_filename)
        .arg(png_filename)
        .status()
        .expect("Failed to execute typst.");
    assert!(status.success());

    std::fs::remove_file(&typst_filename).expect("Could not delete typst file.")
}

fn blur_and_place(
    png_in: &PathBuf,
    png_out: &PathBuf,
    total_size: i32,
    size: i32,
    x: i32,
    y: i32,
    angle: i32,
    thickening: i32,
    blur: i32,
) -> bool {
    // total_size, size, x, y such that x.abs() + size / 2, y.abs() + size / 2 <= total_size / 2
    let displacement = match (x.is_negative(), y.is_negative()) {
        (false, false) => format!("+{}+{}", x, y),
        (false, true) => format!("+{}-{}", x, -y),
        (true, false) => format!("-{}+{}", -x, y),
        (true, true) => format!("-{}-{}", -x, -y),
    };
    let output = Command::new("convert")
        .arg(png_in) // Input image
        .arg("-trim") // Crop
        .arg("+repage")
        .arg("-gravity") // Positions relative to center
        .arg("center")
        .arg("-background") // Set background color
        .arg("white")
        .arg("-rotate") // Rotate (this needs to happen before the other operations)
        .arg(format!("{angle}"))
        .arg("-resize") // Scale image down
        .arg(format!("{size}x{size}"))
        .arg("-extent") // Put image on bigger image with some relative displacement from center
        .arg(format!("{total_size}x{total_size}{displacement}"))
        .arg("-morphology") // Make lines thicker (0.5 but not 0 implements the identity)
        .arg("Erode")
        .arg(format!("Disk:{thickening}.5"))
        .arg("-blur") // Apply Gaussian blur
        .arg(format!("0x{blur}"))
        .arg(png_out) // Output image
        .output()
        .expect("Failed to execute convert.");
    output.stderr.is_empty()
}

// Generate JSON database

fn output_json(data: &Vec<TypstUnicode>, filename: &PathBuf) {
    let json_string = serde_json::to_string(data).unwrap();
    std::fs::write(filename, json_string).expect("Unable to write file.");
}

// Parse and postprocess

#[derive(Debug, PartialEq, Clone, Serialize)]
struct TypstUnicode {
    sym_name: String,
    #[serde(skip_serializing)]
    sym_str: String,
    sym_uni: String,
}

impl TypstUnicode {
    fn new(sym_name: String, sym_str: String) -> Self {
        let sym_uni = sym_str
            .chars()
            .map(|c| format!("{:x}", c as u32))
            .collect::<Vec<String>>()
            .join("-");
        Self {
            sym_name,
            sym_str,
            sym_uni,
        }
    }
}

fn generate_data(input: &str) -> Vec<TypstUnicode> {
    let mut out: Vec<TypstUnicode> = vec![];
    let mut items = match parse_doc(input).unwrap() {
        (_, TypstData::TypstSub(items)) => items,
        _ => unreachable!(), // At the top level, the parser returns a list.
    };

    // DFS to unfold the characters
    while let Some((sym_name, d)) = items.pop() {
        match d {
            TypstData::TypstChar(sym_str) => {
                out.push(TypstUnicode::new(sym_name, sym_str));
            }
            TypstData::TypstSub(list) => {
                for (sub_sym_name, sub_d) in list {
                    let new_name = if sub_sym_name == "" {
                        sym_name.clone()
                    } else {
                        format!("{sym_name}.{sub_sym_name}")
                    };
                    items.push((new_name, sub_d));
                }
            }
        }
    }
    out
}

// Parsing

#[derive(Debug, PartialEq, Clone)]
enum TypstData {
    TypstChar(String),
    TypstSub(Vec<(String, TypstData)>),
}

impl TypstData {
    fn new(sym_str: &str) -> Self {
        Self::TypstChar(String::from(sym_str))
    }
}

fn parse_doc(input: &str) -> IResult<&str, TypstData> {
    // Parse until symbols! macro and then evaluate the rest.
    let (remaining, _) = tuple((many_till(anychar, tag("symbols!")), comment_whitespace))(input)?;
    parse_list(remaining)
}

fn parse_single(input: &str) -> IResult<&str, TypstData> {
    // 'sym_str' [whitespace] (sym_name set to "")
    // Need to be careful with backslash escapes.
    let char_parse = alt((preceded(char('\\'), anychar), none_of("'")));
    let (remaining, sym_str) = terminated(
        delimited(char('\''), many0(char_parse), char('\'')),
        comment_whitespace,
    )(input)?;
    let sym_str: String = sym_str.iter().collect();
    Ok((remaining, TypstData::new(&sym_str)))
}

fn parse_tuple(input: &str) -> IResult<&str, (String, TypstData)> {
    // sym_name: [whitespace] [parse_single or parse_list] [whitespace]
    let (remaining, (sym_name, _)) = many_till(anychar, char(':'))(input)?;
    let (remaining, d) = delimited(
        comment_whitespace,
        alt((parse_single, parse_list)),
        comment_whitespace,
    )(remaining)?;
    let sym_name: String = sym_name.iter().collect();
    Ok((remaining, (sym_name, d)))
}

fn parse_list(input: &str) -> IResult<&str, TypstData> {
    // List of [parse_single or parse_tuple]
    let (remaining, init) = terminated(one_of("[{"), comment_whitespace)(input)?;
    let term = if init == '[' { ']' } else { '}' };

    let single_noname = map(parse_single, |d| (String::from(""), d));
    let single_tuple = terminated(
        terminated(alt((single_noname, parse_tuple)), comment_whitespace),
        terminated(char(','), comment_whitespace),
    );

    let single_noname_closed = map(parse_single, |d| (String::from(""), d));
    let single_tuple_closed = alt((
        value(None, terminated(char(term), comment_whitespace)),
        map(
            tuple((
                alt((single_noname_closed, parse_tuple)),
                comment_whitespace,
                char(term),
                comment_whitespace,
            )),
            |(d, _, _, _)| Some(d),
        ),
    ));

    // Parse items and if there is no trailing comma, append the extra item.
    let (remaining, (mut items, maybe_item)) =
        many_till(single_tuple, single_tuple_closed)(remaining)?;
    if let Some(item) = maybe_item {
        items.push(item);
    }

    Ok((remaining, TypstData::TypstSub(items)))
}

// Parsing helper

fn comment_whitespace(input: &str) -> IResult<&str, ()> {
    let comment = tuple((tag("//"), not_line_ending, multispace0));
    let (remaining, _) = tuple((multispace0, many0(comment)))(input)?;
    Ok((remaining, ()))
}

// Unit testing

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        assert_eq!(Ok(("", TypstData::new("+"))), parse_single("'+'"));
        assert_eq!(Ok(("", TypstData::new("'"))), parse_single("'\\''"));
        assert_eq!(
            Ok(("", (String::from("alpha"), TypstData::new("α")))),
            parse_tuple("alpha: 'α'")
        );
        assert_eq!(
            Ok(("", (String::from("backslash"), TypstData::new("\\")))),
            parse_tuple("backslash: '\\\\'")
        );
        assert_eq!(
            Ok(("", (String::from("quote"), TypstData::new("'")))),
            parse_tuple("quote: '\\''")
        );
    }

    fn get_big_example() -> TypstData {
        TypstData::TypstSub(vec![
            (
                String::from("lt"),
                TypstData::TypstSub(vec![
                    (String::from(""), TypstData::new("<")),
                    (String::from("curly"), TypstData::new("≺")),
                ]),
            ),
            (
                String::from("gt"),
                TypstData::TypstSub(vec![
                    (String::from(""), TypstData::new(">")),
                    (String::from("curly"), TypstData::new("≻")),
                ]),
            ),
        ])
    }

    #[test]
    fn test_parse_composite() {
        assert_eq!(
            Ok((
                "",
                TypstData::TypstSub(vec![
                    (String::from("gamma"), TypstData::new("γ")),
                    (String::from("delta"), TypstData::new("δ"))
                ])
            )),
            parse_list("[gamma: 'γ', delta: 'δ',]")
        );
        assert_eq!(
            Ok((
                "",
                TypstData::TypstSub(vec![
                    (String::from(""), TypstData::new("β")),
                    (String::from("alt"), TypstData::new("ϐ"))
                ])
            )),
            parse_list("['β', alt: 'ϐ']")
        );
        assert_eq!(
            Ok(("", get_big_example())),
            parse_list("[lt: ['<', curly: '≺',], gt: ['>', curly: '≻']]")
        );
    }

    #[test]
    fn test_parse_doc() {
        assert_eq!(
            Ok(("", get_big_example())),
            parse_doc("symbols![// Comment \n lt: ['<', curly: '≺'], gt: ['>', curly: '≻']]")
        );
    }
}

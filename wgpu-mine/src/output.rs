use crate::core::{property_numerator, Evaluation};
use std::io::Write;

pub struct CsvOutput<W: Write> {
    writer: csv::Writer<W>,
}
impl<W: Write> CsvOutput<W> {
    pub fn new(writer: W) -> Result<Self, String> {
        let mut result = Self {
            writer: csv::WriterBuilder::new().buffer_capacity(65536).from_writer(writer),
        };
        result
            .writer
            .write_record([
                "id",
                "name",
                "team",
                "hp",
                "attack",
                "defense",
                "speed",
                "agility",
                "magic",
                "resistance",
                "wisdom",
                "property",
                "xp",
                "xd",
                "skills",
            ])
            .map_err(|e| e.to_string())?;
        result.flush()?;
        Ok(result)
    }
    pub fn write_record(&mut self, id: u64, name: &str, team: &str, value: &Evaluation) -> Result<(), String> {
        let mut fields = vec![id.to_string(), name.to_owned(), team.to_owned()];
        fields.extend(value.props.iter().map(u32::to_string));
        fields.push((f64::from(property_numerator(&value.props)) / 3.0).to_string());
        fields.push(value.xp.to_string());
        fields.push(value.xd.to_string());
        let mut skills: Vec<_> = value
            .skill_ids
            .iter()
            .zip(&value.skill_freqs)
            .filter(|(_, freq)| **freq != 0)
            .map(|(&id, &freq)| (id, freq))
            .collect();
        skills.sort_unstable_by_key(|item| item.0);
        fields.push(skills.iter().map(|(id, freq)| format!("{id}:{freq}")).collect::<Vec<_>>().join(";"));
        self.writer.write_record(fields).map_err(|e| e.to_string())
    }
    pub fn flush(&mut self) -> Result<(), String> { self.writer.flush().map_err(|e| e.to_string()) }
    pub fn into_inner(self) -> Result<W, String> { self.writer.into_inner().map_err(|e| e.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    fn sample() -> Evaluation {
        Evaluation {
            props: [301, 2, 3, 4, 5, 6, 7, 8],
            skill_ids: [24, 9, 0, 3, 12, 2, 8, 7, 6, 5, 4, 1, 10, 11, 13, 14],
            skill_freqs: [2, 0, 7, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            shadow_numerator: 0,
            xp: 1234.56789012345,
            xd: -12.3456789012345,
        }
    }

    #[test]
    fn every_csv_field_roundtrips_and_skills_stay_one_sorted_column() {
        let header = [
            "id",
            "name",
            "team",
            "hp",
            "attack",
            "defense",
            "speed",
            "agility",
            "magic",
            "resistance",
            "wisdom",
            "property",
            "xp",
            "xd",
            "skills",
        ];
        let mut output = CsvOutput::new(Vec::new()).unwrap();
        let value = sample();
        let fields = ["中文🙂,\"\r\n\0尾", "", "逗号,和双\"引号", "\r\n", "\0", "emoji😀"];
        for (id, name) in fields.iter().enumerate() {
            output.write_record(id as u64, name, fields[fields.len() - id - 1], &value).unwrap();
        }
        let mut no_skills = value.clone();
        no_skills.skill_freqs = [0; 16];
        output.write_record(u64::MAX, "没有技能", "队伍", &no_skills).unwrap();
        output.flush().unwrap();
        let bytes = output.into_inner().unwrap();
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        assert!(reader.headers().unwrap().iter().eq(header));
        let rows = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(rows.len(), fields.len() + 1);
        for (id, row) in rows[..fields.len()].iter().enumerate() {
            assert_eq!(row.len(), 15);
            assert_eq!(row[0].parse::<u64>().unwrap(), id as u64);
            assert_eq!(&row[1], fields[id]);
            assert_eq!(&row[2], fields[fields.len() - id - 1]);
            for index in 0..8 {
                assert_eq!(row[index + 3].parse::<u32>().unwrap(), value.props[index]);
            }
            assert_eq!(
                row[11].parse::<f64>().unwrap(),
                f64::from(property_numerator(&value.props)) / 3.0
            );
            assert_eq!(row[12].parse::<f64>().unwrap(), value.xp);
            assert_eq!(row[13].parse::<f64>().unwrap(), value.xd);
            assert_eq!(&row[14], "0:7;3:4;12:1;24:2");
        }
        assert_eq!(rows.last().unwrap()[0].parse::<u64>().unwrap(), u64::MAX);
        assert_eq!(&rows.last().unwrap()[14], "");
        let empty = CsvOutput::new(Vec::new()).unwrap().into_inner().unwrap();
        let mut reader = csv::Reader::from_reader(empty.as_slice());
        assert!(reader.headers().unwrap().iter().eq(header));
        assert_eq!(reader.records().count(), 0);
    }

    #[derive(Clone, Debug)]
    struct FailWriter(Rc<Cell<u8>>);
    impl Write for FailWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0.get() == 1 {
                Err(std::io::Error::other("模拟写入失败"))
            } else {
                Ok(bytes.len())
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if self.0.get() == 2 {
                Err(std::io::Error::other("模拟刷新失败"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn independent_write_flush_and_into_inner_errors_propagate() {
        for mode in [1, 2] {
            assert!(CsvOutput::new(FailWriter(Rc::new(Cell::new(mode)))).is_err());
        }
        let mode = Rc::new(Cell::new(0));
        let mut output = CsvOutput::new(FailWriter(Rc::clone(&mode))).unwrap();
        mode.set(1);
        let error = output.write_record(1, &"中".repeat(65536), "队伍", &sample()).unwrap_err();
        assert!(error.contains("模拟写入失败"));
        mode.set(0);
        drop(output);
        let mut output = CsvOutput::new(FailWriter(Rc::clone(&mode))).unwrap();
        output.write_record(2, "中文\0", "队伍", &sample()).unwrap();
        mode.set(1);
        assert!(output.flush().unwrap_err().contains("模拟写入失败"));
        mode.set(0);
        output.flush().unwrap();
        mode.set(2);
        assert!(output.flush().unwrap_err().contains("模拟刷新失败"));
        assert!(output.into_inner().unwrap_err().contains("模拟刷新失败"));
        mode.set(0);
    }
    #[test]
    fn csv_roundtrip_and_zero_hits() {
        let mut output = CsvOutput::new(Vec::new()).unwrap();
        let name = "中文,双引号".to_owned()
            + &char::from(34).to_string()
            + "换行"
            + &char::from(13).to_string()
            + &char::from(10).to_string()
            + "😀";
        let value = crate::core::evaluate("team", &name);
        output.write_record(9, &name, "队,伍", &value).unwrap();
        let bytes = output.into_inner().unwrap();
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        assert_eq!(reader.headers().unwrap().len(), 15);
        let rows: Vec<_> = reader.records().collect::<Result<_, _>>().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(&rows[0][1], name);
        assert_eq!(&rows[0][2], "队,伍");
        let empty = CsvOutput::new(Vec::new()).unwrap().into_inner().unwrap();
        assert_eq!(csv::Reader::from_reader(empty.as_slice()).records().count(), 0);
    }
    #[test]
    fn writer_errors_propagate() {
        struct Fail;
        impl Write for Fail {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(std::io::Error::other("write failed")) }
            fn flush(&mut self) -> std::io::Result<()> { Err(std::io::Error::other("flush failed")) }
        }
        assert!(CsvOutput::new(Fail).is_err());
    }
}

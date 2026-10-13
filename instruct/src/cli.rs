use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about)]
pub struct Args {
    #[arg(short, long)]
    pub model_file: PathBuf,
}

pub async fn run_model(args: &Args) -> anyhow::Result<()> {
    println!("loading... {}", args.model_file.display());
    Ok(())
}

use instruct::cli::{Args, run_model};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_model(&args).await?;
    Ok(())
}

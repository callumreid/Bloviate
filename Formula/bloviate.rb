class Bloviate < Formula
  desc "Voice-fingerprinting dictation tool for whispering in noisy environments"
  homepage "https://github.com/callumreid/Bloviate"
  license "MIT"
  head "https://github.com/callumreid/Bloviate.git", branch: "main"

  depends_on "portaudio"
  depends_on "python@3.12"

  preserve_rpath

  def install
    venv = libexec/"venv"
    python = Formula["python@3.12"].opt_bin/"python3.12"

    system python, "-m", "venv", venv
    system venv/"bin/python", "-m", "pip", "install", "--upgrade", "pip"
    system venv/"bin/python", "-m", "pip", "install", "--no-cache-dir", "."

    bin.write_exec_script venv/"bin/bloviate"
  end

  test do
    assert_match "--doctor", shell_output("#{bin}/bloviate --help")
  end
end

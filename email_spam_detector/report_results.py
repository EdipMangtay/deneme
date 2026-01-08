"""
Report fine-tuning and training results.
"""
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def report_fine_tuning_results():
    """Report fine-tuning results from terminal output."""
    logger.info("\n" + "=" * 70)
    logger.info("📊 FINE-TUNING SONUÇLARI")
    logger.info("=" * 70)
    
    logger.info("\n✅ Fine-tuning başarıyla tamamlandı!")
    logger.info("\n📈 Evaluation Sonuçları (Test Set):")
    logger.info("   • Accuracy: 97.92%")
    logger.info("   • F1 Score: 97.91%")
    logger.info("   • Loss: 0.067")
    logger.info("\n💾 Model Checkpoint:")
    logger.info("   • Konum: artifacts/checkpoint/checkpoint/checkpoint-1008")
    logger.info("   • Epoch: 2.0")
    logger.info("   • Training Samples: 11,517")
    logger.info("\n📦 Dataset Bilgileri:")
    logger.info("   • w1998: 5,728 emails")
    logger.info("   • abdallah: 5,572 emails")
    logger.info("   • kucev: 84 emails")
    logger.info("   • Gmail: 599 emails (500 inbox + 99 spam)")
    logger.info("   • Toplam: 11,517 emails")
    logger.info("   • Label dağılımı: 9,390 NOT SPAM (0) + 2,127 SPAM (1)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ FINE-TUNING BAŞARIYLA TAMAMLANDI!")
    logger.info("=" * 70)


if __name__ == "__main__":
    report_fine_tuning_results()



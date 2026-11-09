#include "ember/char_model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using ember::CharModel;
using ember::CharModelConfig;
using ember::CharTrainer;

CharModelConfig Tiny(std::size_t vocab) {
  CharModelConfig c;
  c.vocab = vocab;
  c.context = 6;
  c.width = 8;
  c.layers = 2;
  c.heads = 2;
  c.mlp = 12;
  return c;
}

// The whole model's gradient against finite differences, for every parameter, in double.
TEST(CharModel, TheWholeModelsGradientsMatchFiniteDifferences) {
  const CharModel<double> model(Tiny(5), 3);
  const std::vector<std::int32_t> tokens{0, 3, 1, 4, 4, 2, 0};
  ember::autograd::ZeroGrad(model.parameters());
  ember::autograd::Backward(model.Loss(tokens));
  const double h = 1e-6;
  std::size_t checked = 0;
  for (const auto& p : model.parameters()) {
    for (std::size_t i = 0; i < p->value.size(); ++i) {
      const double saved = p->value[i];
      p->value[i] = saved + h;
      const double up = model.Loss(tokens)->value[0];
      p->value[i] = saved - h;
      const double down = model.Loss(tokens)->value[0];
      p->value[i] = saved;
      const double numeric = (up - down) / (2 * h);
      ASSERT_NEAR(p->grad[i], numeric, 1e-6 * std::max(1.0, std::abs(numeric)));
      ++checked;
    }
  }
  EXPECT_EQ(checked, model.parameter_count());
}

TEST(CharModel, StartsNearTheLossOfGuessing) {
  const CharModel<float> model(Tiny(20), 1);
  const std::vector<std::int32_t> tokens{1, 5, 9, 13, 17, 2, 6};
  // An untrained model should be close to uniform: log(20) nats per character.
  EXPECT_NEAR(model.Loss(tokens)->value[0], std::log(20.0F), 0.5F);
}

TEST(CharTrainer, LearnsARepetitiveText) {
  std::string text;
  while (text.size() < 4000) text += "the cat sat on the mat. ";
  CharTrainer::Options options;
  options.model.context = 16;
  options.model.width = 32;
  options.model.layers = 1;
  options.model.heads = 2;
  options.model.mlp = 64;
  options.batch = 4;
  options.rate = 1e-2F;
  CharTrainer trainer(text, options);
  ASSERT_TRUE(trainer.ready());
  const float before = trainer.HeldOutLoss(4);
  for (int step = 0; step < 150; ++step) trainer.Step();
  const float after = trainer.HeldOutLoss(4);
  // Text it never trained on, and still nearly certain of every next character.
  EXPECT_LT(after, before / 4) << "before " << before << ", after " << after;
  EXPECT_EQ(trainer.steps(), 150U);
  const std::string sample = trainer.Sample("the ", 20, 0.1F, 7);
  EXPECT_EQ(sample.size(), 24U);
  // After "the " the text only ever has "cat" or "mat".
  const std::string next = sample.substr(4, 4);
  EXPECT_TRUE(next == "cat " || next == "mat.") << sample;
}

TEST(CharTrainer, SamplingIsRepeatableWithASeed) {
  std::string text;
  while (text.size() < 1000) text += "abcdefg hijk ";
  CharTrainer::Options options;
  options.model = Tiny(0);
  options.model.context = 8;
  CharTrainer trainer(text, options);
  EXPECT_EQ(trainer.Sample("ab", 30, 1.0F, 5), trainer.Sample("ab", 30, 1.0F, 5));
  EXPECT_EQ(trainer.vocab().size(), 12U);
}

}  // namespace

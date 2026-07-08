/**
 * Клієнтська валідація форми передбачення діабету.
 * Дублює серверні діапазони з config.VALID_RANGES для швидкого зворотного зв'язку.
 * Стиль коду: StandardJS (const/let, без крапок з комою).
 */
'use strict'

;(function () {
  /** Правила валідації числових полів (синхронізовано з config.VALID_RANGES). */
  const FIELD_RULES = {
    age: { min: 1, max: 120, label: 'Вік' },
    bmi: { min: 10, max: 80, label: 'ІМТ' },
    HbA1c_level: { min: 3, max: 15, label: 'HbA1c' },
    blood_glucose_level: { min: 50, max: 400, label: 'Глюкоза в крові' }
  }

  /**
   * Перевіряє одне числове поле форми.
   *
   * @param {HTMLInputElement} input - Поле вводу.
   * @param {{min: number, max: number, label: string}} rule - Правило валідації.
   * @returns {string|null} Текст помилки або null, якщо поле валідне.
   */
  function validateNumberField (input, rule) {
    const value = Number(input.value)

    if (input.value === '' || Number.isNaN(value)) {
      return rule.label + ' має бути числом.'
    }

    if (value < rule.min || value > rule.max) {
      return rule.label + ' має бути в діапазоні ' + rule.min + '–' + rule.max + '.'
    }

    return null
  }

  /**
   * Показує або прибирає повідомлення про помилку біля поля.
   *
   * @param {HTMLInputElement} input - Поле вводу.
   * @param {string|null} message - Текст помилки.
   */
  function setFieldError (input, message) {
    const errorId = input.id + '-error'
    let existing = document.getElementById(errorId)

    if (message) {
      input.setAttribute('aria-invalid', 'true')
      if (!existing) {
        existing = document.createElement('span')
        existing.id = errorId
        existing.className = 'field-error'
        existing.setAttribute('role', 'alert')
        input.insertAdjacentElement('afterend', existing)
      }
      existing.textContent = message
      return
    }

    input.removeAttribute('aria-invalid')
    if (existing) {
      existing.remove()
    }
  }

  /**
   * Валідує всю форму перед відправкою на сервер.
   *
   * @param {HTMLFormElement} form - Форма передбачення.
   * @returns {boolean} true, якщо всі поля коректні.
   */
  function validateForm (form) {
    let isValid = true

    Object.keys(FIELD_RULES).forEach(function (fieldName) {
      const input = form.elements.namedItem(fieldName)
      if (!(input instanceof HTMLInputElement)) {
        return
      }

      const errorMessage = validateNumberField(input, FIELD_RULES[fieldName])
      setFieldError(input, errorMessage)
      if (errorMessage) {
        isValid = false
      }
    })

    return isValid
  }

  document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form')
    if (!form) {
      return
    }

    form.addEventListener('submit', function (event) {
      if (!validateForm(form)) {
        event.preventDefault()
      }
    })

    Object.keys(FIELD_RULES).forEach(function (fieldName) {
      const input = form.elements.namedItem(fieldName)
      if (input instanceof HTMLInputElement) {
        input.addEventListener('input', function () {
          setFieldError(input, validateNumberField(input, FIELD_RULES[fieldName]))
        })
      }
    })

    const thresholdSlider = document.getElementById('prediction_threshold')
    const thresholdDisplay = document.getElementById('threshold-display')
    if (thresholdSlider instanceof HTMLInputElement && thresholdDisplay) {
      const updateThresholdDisplay = function () {
        const value = thresholdSlider.value
        thresholdDisplay.textContent = value + '%'
        thresholdSlider.setAttribute('aria-valuenow', value)
        thresholdSlider.setAttribute('aria-valuetext', value + ' відсотків')
      }
      thresholdSlider.addEventListener('input', updateThresholdDisplay)
      updateThresholdDisplay()
    }
  })
}())
